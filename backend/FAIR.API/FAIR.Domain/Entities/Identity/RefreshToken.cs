using System.ComponentModel.DataAnnotations;

namespace FAIR.Domain.Entities.Identity
{
    public class RefreshToken
    {
        [Key]
        public string Id { get; set; } = Guid.NewGuid().ToString();
        public string UserId { get; set; } = string.Empty;
        public string Token { get; set; } = string.Empty;

    }
}
